// Removed Claim type import as it's not needed in JavaScript
// Suppression de l'importation du type Claim, car elle n'est pas nécessaire en JavaScript
const BASE_URL = '/api/claims';

export async function fetchClaims(tag) {
  const url = tag
    ? `${BASE_URL}?tag=${encodeURIComponent(tag)}`
    : BASE_URL;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error('Failed to fetch claims', { cause: response });
  }

  return response.json();
}


// ======================================================
// ML PREDICTION
// ======================================================

export async function predictClaimTag(userClaim) {

  const response = await fetch(
    'http://127.0.0.1:8000/predict',
    {
      method: 'POST',

      headers: {
        'Content-Type': 'application/json',
      },

      body: JSON.stringify({
        user_claim: userClaim,
      }),
    }
  );

  if (!response.ok) {
    throw new Error('Prediction failed');
  }

  return response.json();
}

export async function updateClaimTag(claimId, tag) {
  const response = await fetch(`${BASE_URL}/${claimId}/tag`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ tag }),
  });

  if (!response.ok) {
    throw new Error('Failed to update tag');
  }
}
